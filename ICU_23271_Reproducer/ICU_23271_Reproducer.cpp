#include "unicode/unistr.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#define SIMBASSERT(cond) do { if (!(cond)) { std::abort(); } } while(0)

namespace Simba
{
namespace Support
{
    /// @brief
    class ThreadSafeRefCount
    {
        ThreadSafeRefCount(const ThreadSafeRefCount&) = delete;
        ThreadSafeRefCount& operator=(const ThreadSafeRefCount&) = delete;
    // Public ===============================================================================================================
    public:
        /// @brief Constructor which initializes the reference count to 0.
        ThreadSafeRefCount() noexcept = default;

        /// @brief Constructor which initializes the reference count to the given value
        explicit ThreadSafeRefCount(uint32_t in_initialCount) noexcept : m_refCount(in_initialCount)
        {
            // Do nothing.
        }

        /// @brief Destructor.
        ~ThreadSafeRefCount()
        {
            // Can't throw from destructor,
            // (& this indicates a _serious_ problem, probably memory corruption or use-after-free, or sets up a future
            // use-after-free)
            SIMBASSERT(!m_refCount.load(std::memory_order_relaxed));
        }

        bool HasReferences() const noexcept
        {
            return GetShareState() != NO_REFERENCES;
        }

        enum ShareState : int
        {
            NO_REFERENCES,
            SINGLE_REFERENCE,
            MULTIPLE_REFERENCES
        };

        ShareState GetShareState() const
        {
            const auto count(m_refCount.load(std::memory_order_acquire));
            if (!count)
            {
                return NO_REFERENCES;
            }
            else if (count == 1)
            {
                return SINGLE_REFERENCE;
            }

            return MULTIPLE_REFERENCES;
        }

        /// @brief Unshare the object, setting the reference count to 0.
        ///
        /// Must only be called when GetShareState() == SINGLE_REFERENCE
        void Unshare()
        {
            const auto state(GetShareState());
            SIMBASSERT(SINGLE_REFERENCE == state);

            m_refCount.store(0, std::memory_order_relaxed);
        }

        /// @brief Should only be called on a subclass instance which has never been used with a SharedPtr.
        ///
        /// Useful if you _know_ the initial reference count, to avoid extra atomic operations.
        void OverrideInitialReferenceCount(uint32_t in_initialCount = 1) noexcept
        {
            // Can't throw from noexcept function
            // (& throwing at this point would probably cause a leak or future crash in the destructor)
            SIMBASSERT(!HasReferences());
            m_refCount = in_initialCount;
        }

        /// @brief Increase the reference count by 1.
        ///
        /// This is part of the API to "qualify" as a shared object to be managed by SharedPtr.
        /// This implementation is thread-safe.
        void Retain() noexcept
        {
            // Can't throw from noexcept function
            // (& this indicates a _serious_ problem, probably memory corruption or use-after-free)
            SIMBASSERT(m_refCount.fetch_add(1u, std::memory_order_relaxed) + 1 != 0);
        }

        /// @brief Decrease the reference count by 1.
        ///
        /// @return Whether the reference count reached 0.
        bool Release() noexcept
        {
            const auto previousCount(m_refCount.fetch_sub(1u, std::memory_order_release));
            if (previousCount == 1)
            {
                std::atomic_thread_fence(std::memory_order_acquire);
                return true;
            }

            SIMBASSERT(previousCount);

            return false;
        }
    // Private ==============================================================================================================
    private:
        // The reference count to be used as a shared object.
        std::atomic_uint m_refCount = { 0 };
    };
}
}

namespace Simba
{
namespace Support
{
namespace Impl
{
    // Gets the first type in a parameter pack.
    template <typename ...Ts>
    struct FirstElement
    {
        using type = typename std::tuple_element<0, std::tuple<Ts...>>::type;
    };

    template <>
    struct FirstElement<>
    {
        using type = void;
    };

    // Class which nothing should 'really' inherit from
    struct UniqueThreadSafeSharedObjectTBaseClass {};
}

    /// @brief This template class provides a thread-safe implementation of SharedObject interface.
    template <typename BaseType>
    class ThreadSafeSharedObjectT : public BaseType
    {
        template <typename T>
        friend class SharedPtr;

    // Public ===============================================================================================================
    public:
        /// @brief Default constructor.
        ThreadSafeSharedObjectT() = default;

        bool HasReferences() const noexcept
        {
            return m_refCount.HasReferences();
        }

        /// @brief Used to manually control the reference count (i.e. outside of SharedPtr). Equivalent to Retain()
        void ManualRetain() const noexcept
        {
            Retain();
        }

        /// @brief Used to manually control the reference count (i.e. outside of SharedPtr). Equivalent to Release()
        void ManualRelease() const noexcept
        {
            Release();
        }

    // Protected ============================================================================================================
    protected:
        /// @brief Constructor. Forwards all arguments to T's constructor.
        template <
            typename ...Params,
            // Disable this constructor when the copy/move constructor should be used instead
            std::enable_if_t<(sizeof...(Params) != 1) ||
            !std::is_convertible_v<typename Simba::Support::Impl::FirstElement<Params...>, const BaseType&>> = 0>
        ThreadSafeSharedObjectT(Params&&... params) : BaseType(std::forward<Params>(params)...)
        {
            // Do nothing.
        }

        /// @brief Copy Constructor. Does NOT copy the reference count.
        ThreadSafeSharedObjectT(const BaseType& in_other) : BaseType(in_other)
        {
            // Do nothing.
        }

        /// @brief Move Constructor. Does NOT copy/move/mutate the reference count.
        ThreadSafeSharedObjectT(BaseType&& in_other) noexcept(noexcept(BaseType(in_other))) :
            BaseType(std::move(in_other))
        {
            // Do nothing.
        }

        /// @brief Assignment operator. Does NOT copy the reference count.
        ThreadSafeSharedObjectT& operator=(const ThreadSafeSharedObjectT& in_other)
        {
            static_cast<BaseType&>(*this) = static_cast<const BaseType&>(in_other);
            return *this;
        }

        /// @brief Move-assignment operator. Does NOT copy/move/mutate the reference count.
        ThreadSafeSharedObjectT& operator=(ThreadSafeSharedObjectT&& in_other)
        {
            static_cast<BaseType&>(*this) = std::move(static_cast<BaseType&&>(in_other));
            return *this;
        }

        /// @brief Assignment operator. Does NOT copy the reference count.
        ThreadSafeSharedObjectT& operator=(const BaseType& in_other)
        {
            static_cast<BaseType&>(*this) = in_other;
            return *this;
        }

        /// @brief Move-assignment operator. Does NOT copy/move/mutate the reference count.
        ThreadSafeSharedObjectT& operator=(BaseType&& in_other)
        {
            static_cast<BaseType&>(*this) = std::move(in_other);
            return *this;
        }

        /// @brief Destructor.
        ///
        /// The destructor is protected to prevent client code from explicitly using delete instead
        /// of calling Release() to "delete" the object.
        virtual ~ThreadSafeSharedObjectT() = default;

        ThreadSafeRefCount::ShareState GetShareState() const
        {
            return m_refCount.GetShareState();
        }

        /// @brief Unshare the object, setting the reference count to 0.
        ///
        /// Must only be called when GetShareState() == SINGLE_REFERENCE
        void Unshare()
        {
            m_refCount.Unshare();
        }

        /// @brief Called when the reference count reaches 0.
        virtual void OnFinalRelease() const /* noexcept (Will be changed in a future major version) */
        {
            delete this;
        }

        /// @brief Should only be called on a subclass instance which has never been used with a SharedPtr.
        ///
        /// Useful if you _know_ the initial reference count, to avoid extra atomic operations.
        void OverrideInitialReferenceCount(uint32_t in_initialCount = 1) noexcept
        {
            m_refCount.OverrideInitialReferenceCount(in_initialCount);
        }

    // Private ==============================================================================================================
    private:
        /// @brief Increase the reference count by 1.
        ///
        /// This is part of the API to "qualify" as a shared object to be managed by SharedPtr.
        /// This implementation is thread-safe.
        ///
        /// WARNING:
        /// This method is designed to be used by SharedPtr only. Never call this method 
        /// directly.
        void Retain() const noexcept
        {
            m_refCount.Retain();
        }

        /// @brief Decrease the reference count by 1.
        ///
        /// This is part of the API to "qualify" as a shared object to be managed by SharedPtr.
        /// This implementation is thread-safe.
        ///
        /// WARNING:
        /// This method is designed to be used by SharedPtr only. Never call this method 
        /// directly.
        void Release() const noexcept
        {
            if (m_refCount.Release())
            {
                OnFinalRelease();
            }
        }

        // The reference count to be used as a shared object.
        mutable ThreadSafeRefCount m_refCount;
    };

    using ThreadSafeSharedObject = ThreadSafeSharedObjectT<Impl::UniqueThreadSafeSharedObjectTBaseClass>;
}
}

namespace Simba
{
namespace Support
{
    class simba_wstring_impl final : public ThreadSafeSharedObjectT<U_ICU_NAMESPACE::UnicodeString>
    {
        using BaseType = ThreadSafeSharedObjectT<U_ICU_NAMESPACE::UnicodeString>;
    // Public ===============================================================================================================
    public:
        simba_wstring_impl() = default;

        /// @brief Constructor. Forwards all arguments to T's constructor.
        template <
            typename ...Params,
            // Disable this constructor when the copy/move constructor should be used instead
            std::enable_if_t<(sizeof...(Params) != 1) ||
                !std::is_convertible_v<typename Simba::Support::Impl::FirstElement<Params...>, const BaseType&>> = 0>
        simba_wstring_impl(Params&&... params) : BaseType(std::forward<Params>(params)...)
        {
            // Do nothing.
        }

        simba_wstring_impl(const simba_wstring_impl& in_other) :
            BaseType(static_cast<const U_ICU_NAMESPACE::UnicodeString&>(in_other))
        {
            // Do nothing.
        }

        simba_wstring_impl(U_ICU_NAMESPACE::UnicodeString&& in_other) : BaseType(std::move(in_other))
        {
            // Do nothing.
        }

        simba_wstring_impl(const U_ICU_NAMESPACE::UnicodeString& in_other) : BaseType(in_other)
        {
            // Do nothing.
        }

        void TriggerError()
        {
            *this += *this;
        }

        using BaseType::BaseType;
        using BaseType::operator=;

        using BaseType::GetShareState;
        using BaseType::OverrideInitialReferenceCount;
        using BaseType::Unshare;
    };
}
}

int main()
{
    return EXIT_SUCCESS;
}